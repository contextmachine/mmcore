# Closest-Point Valley-Marching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `bez_surface_closest_points` resolve degenerate "valley" configurations (point near the axis of a surface of revolution / center of an ellipsoid) by tracing the valley floor instead of subdividing it, eliminating the `max_cells` blow-up while finding all minima.

**Architecture:** Keep the subdivision backbone. When a cell, after several subdivisions, is still unpruned and its cross-valley curvature is certified positive-definite by a Bernstein-hull test (exactly one of `g_uu`/`g_vv` is hull-positive), hand it to a marcher that traces the floor `∂g/∂(corr)=0` across the cell and reports the along-valley `∂g/∂(march)` sign-changes as minima — then drop the cell (provably covered). A relative `ptol` floor is a complementary termination safeguard.

**Tech Stack:** Python 3.9+, NumPy. Extends `mmcore/numeric/_bez_closest_point.py`. Tests via `.venv/bin/python -m pytest`. Spec: `docs/superpowers/specs/2026-06-28-closest-point-valley-marching-design.md`.

---

## File Structure

- **Modify** `mmcore/numeric/_bez_closest_point.py` — all new code is added here (the closest-point module), in a clearly-marked `# --- valley marching ---` section, plus a small edit to the `bez_surface_closest_points` cell loop and the two `ptol` clamps.
- **Modify** `tests/test_bez_closest_point.py` — new tests appended.
- **Do NOT modify** `mmcore/numeric/closest_point.py` (legacy baseline).

Conventions (unchanged): a surface net `S` is homogeneous `(m+1, n+1, 4)` when `rational`, else `(m+1, n+1, 3)`. Scalar Bernstein nets carry no trailing value axis; `_deriv_net`/`_split_net` wrap the trailing-axis handling. `axis=0` is `u`, `axis=1` is `v`. Existing reusable helpers: `_deriv_net(net, axis)`, `_bernstein_product_nd(a, b)`, `_hull_excludes_zero(net)`, `_split_net(net, axis, t=0.5)`, `eval_surface_d2(S, u, v, rational)`, `eval_surface(S, u, v, rational)`, `newton_surface_closest_point(S, P, u0, v0, rational=, bounds=)`, `_classify_surface_min(S, P, u, v, rational, atol)`, `_dedup_add(out, u, v, dist, pt, kind, ptol_u, ptol_v, atol)`, `point_surface_stationarity_nets(P, S, rational)`.

Note `eval_surface_d2` returns `(pt, Su, Sv, Suu, Suv, Svv)` (Task 3 of the original plan).

---

### Task 1: Pointwise squared-distance derivatives `_surface_g_derivs`

**Files:**
- Modify: `mmcore/numeric/_bez_closest_point.py`
- Test: `tests/test_bez_closest_point.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bez_closest_point.py  (append)
from mmcore.numeric._bez_closest_point import _surface_g_derivs


def test_surface_g_derivs_matches_finite_difference():
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.5]],
                  [[1.0, 0.0, 0.5], [1.0, 1.0, 0.0]]])
    P = np.array([0.4, 0.6, 2.0])
    u, v = 0.37, 0.62
    g, gu, gv, guu, guv, gvv = _surface_g_derivs(S, P, u, v, rational=False)

    def gfun(uu, vv):
        s = eval_surface(S, uu, vv, rational=False) - P
        return float(np.dot(s, s))
    h = 1e-5
    assert abs(g - gfun(u, v)) < 1e-9
    assert abs(gu - (gfun(u + h, v) - gfun(u - h, v)) / (2 * h)) < 1e-3
    assert abs(gv - (gfun(u, v + h) - gfun(u, v - h)) / (2 * h)) < 1e-3
    assert abs(gvv - (gfun(u, v + h) - 2 * gfun(u, v) + gfun(u, v - h)) / h**2) < 1e-1
    assert abs(guv - (gfun(u + h, v + h) - gfun(u + h, v - h)
                      - gfun(u - h, v + h) + gfun(u - h, v - h)) / (4 * h**2)) < 1e-1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -k surface_g_derivs -v`
Expected: FAIL — `ImportError: cannot import name '_surface_g_derivs'`.

- [ ] **Step 3: Write minimal implementation**

```python
# mmcore/numeric/_bez_closest_point.py  (append, under a new section)
# ---------------------------------------------------------------------------
# Valley marching: trace a degenerate sq-distance valley instead of
# subdividing it. See docs/.../2026-06-28-closest-point-valley-marching-design.md
# ---------------------------------------------------------------------------

def _surface_g_derivs(S, point, u, v, rational):
    """Pointwise g = ||S(u,v)-point||^2 and its 1st/2nd derivatives.

    Returns (g, g_u, g_v, g_uu, g_uv, g_vv).
    """
    pt, su, sv, suu, suv, svv = eval_surface_d2(S, u, v, rational=rational)
    d = pt - point
    g = float(np.dot(d, d))
    gu = 2.0 * float(np.dot(d, su))
    gv = 2.0 * float(np.dot(d, sv))
    guu = 2.0 * (float(np.dot(su, su)) + float(np.dot(d, suu)))
    guv = 2.0 * (float(np.dot(su, sv)) + float(np.dot(d, suv)))
    gvv = 2.0 * (float(np.dot(sv, sv)) + float(np.dot(d, svv)))
    return g, gu, gv, guu, guv, gvv
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -k surface_g_derivs -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/_bez_closest_point.py tests/test_bez_closest_point.py
git commit -m "feat(closest-point): add pointwise squared-distance derivatives helper"
```

---

### Task 2: Cross-valley curvature sign net `_g_diag_sign_net`

**Files:**
- Modify: `mmcore/numeric/_bez_closest_point.py`
- Test: `tests/test_bez_closest_point.py`

The certificate needs a Bernstein net whose sign equals `sign(∂²g/∂a²)` over a cell. With `g = F/w²` and the stationarity net `N_a = F_a·w − 2F·w_a`, one has `∂²g/∂a² = (N_{a,a}·w − 3·N_a·w_a)/w⁴`, so the sign net is `M_aa = N_{a,a}·w − 3·N_a·w_a` (a Bernstein net; for non-rational `w≡1` it reduces to `N_{a,a}`). `M_aa.min() > 0` (Bernstein convex-hull property) proves `∂²g/∂a² > 0` everywhere in the cell.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bez_closest_point.py  (append)
from mmcore.numeric._bez_closest_point import (
    _g_diag_sign_net, point_surface_stationarity_nets, _surface_g_derivs)
from math import comb


def _eval_bern_2d_local(net, u, v):  # local copy: scalar bivariate Bernstein eval
    mm, nn = net.shape[0] - 1, net.shape[1] - 1
    Bu = np.array([comb(mm, i) * u**i * (1 - u) ** (mm - i) for i in range(mm + 1)])
    Bv = np.array([comb(nn, j) * v**j * (1 - v) ** (nn - j) for j in range(nn + 1)])
    return float(Bu @ net @ Bv)


def test_g_diag_sign_net_sign_matches_gvv_rational():
    # Rational patch (sphere-octant-like): sign of M_vv net must equal sign of g_vv.
    s = np.sqrt(2) / 2
    cp = np.array([[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                   [[1, 0, 1], [1, 1, 1], [0, 1, 1]],
                   [[1, 0, 0], [1, 1, 0], [0, 1, 0]]], dtype=float)
    w = np.array([[1.0, s, 1.0], [s, 0.5, s], [1.0, s, 1.0]])
    H = np.concatenate([cp * w[:, :, None], w[:, :, None]], axis=2)
    P = np.array([0.3, 0.2, 0.4])
    Nu, Nv, F, Sw = point_surface_stationarity_nets(P, H, rational=True)
    Mvv = _g_diag_sign_net(Nv, Sw, axis=1)
    for u in (0.2, 0.5, 0.8):
        for v in (0.25, 0.6, 0.85):
            _, _, _, _, _, gvv = _surface_g_derivs(H, P, u, v, rational=True)
            assert np.sign(_eval_bern_2d_local(Mvv, u, v)) == np.sign(gvv)


def test_g_diag_sign_net_nonrational_is_second_derivative():
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 2.0, 0.0]],
                  [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]]])
    P = np.array([1.0, 1.0, 3.0])
    Nu, Nv, F, Sw = point_surface_stationarity_nets(P, S, rational=False)
    Mvv = _g_diag_sign_net(Nv, Sw, axis=1)
    for u in (0.3, 0.7):
        for v in (0.3, 0.7):
            _, _, _, _, _, gvv = _surface_g_derivs(S, P, u, v, rational=False)
            assert np.sign(_eval_bern_2d_local(Mvv, u, v)) == np.sign(gvv)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -k g_diag_sign_net -v`
Expected: FAIL — `ImportError: cannot import name '_g_diag_sign_net'`.

- [ ] **Step 3: Write minimal implementation**

```python
# mmcore/numeric/_bez_closest_point.py  (append in the valley section)

def _g_diag_sign_net(N_diag, Sw, axis):
    """Bernstein net whose sign equals sign(d^2 g / d{axis}^2) over the cell.

    M_aa = N_{a,a}*w - 3*N_a*w_a, where N_diag = N_u (axis 0) or N_v (axis 1)
    and Sw is the (cell-local) weight net. For non-rational input w == 1 so
    w_a == 0 and this reduces to the second-derivative net N_{a,a}.
    """
    Ndd = _deriv_net(N_diag, axis)
    Swd = _deriv_net(Sw, axis)
    return _bernstein_product_nd(Ndd, Sw) - 3.0 * _bernstein_product_nd(N_diag, Swd)


def _hull_positive(net):
    """True if the Bernstein hull proves the polynomial is > 0 over the cell."""
    return float(net.min()) > 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -k g_diag_sign_net -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/_bez_closest_point.py tests/test_bez_closest_point.py
git commit -m "feat(closest-point): add cross-valley curvature sign net"
```

---

### Task 3: Valley-floor corrector `_valley_floor_solve`

**Files:**
- Modify: `mmcore/numeric/_bez_closest_point.py`
- Test: `tests/test_bez_closest_point.py`

Given a fixed march-axis coordinate, solve `∂g/∂(corr)=0` for the corrector coordinate via bounded 1-D Newton. Returns `(corr_value, ok)` where `ok` means: converged to a strictly-interior point with positive cross-valley curvature (a genuine valley-floor point), not captured at a boundary.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bez_closest_point.py  (append)
from mmcore.numeric._bez_closest_point import _valley_floor_solve


def _ruled_U_surface(H=4.0):
    # A cubic "U" curve in z=0 extruded in +z by H -> a ruled surface with a
    # clean v-valley (linear in v => g_vv constant > 0) and 2 u-minima.
    cu = np.array([[-2.0, 2.0, 0.0], [-2.0, -3.0, 0.0], [2.0, -3.0, 0.0], [2.0, 2.0, 0.0]])
    net = np.zeros((4, 2, 3))
    net[:, 0, :] = cu
    net[:, 1, :] = cu + np.array([0.0, 0.0, H])
    return net


def test_valley_floor_solve_finds_interior_floor():
    S = _ruled_U_surface(H=4.0)
    P = np.array([0.0, 1.0, 2.0])     # mid-height -> floor at v=0.5 for all u
    # march axis = u (0); correct in v (1)
    c, ok = _valley_floor_solve(S, P, march_val=0.5, corr_seed=0.5,
                                corr_lo=0.0, corr_hi=1.0, march_axis=0,
                                rational=False, ctol=1e-9)
    assert ok
    assert abs(c - 0.5) < 1e-6   # v* = 0.5 (P's height)


def test_valley_floor_solve_reports_not_ok_off_floor():
    # A patch where, at this march value, dg/dv has no interior zero in [v0,v1]
    # (the cross-valley min lies on the v=0 edge) -> ok must be False.
    S = _ruled_U_surface(H=4.0)
    P = np.array([0.0, 1.0, -5.0])    # below the patch -> closest v is the v=0 edge
    c, ok = _valley_floor_solve(S, P, march_val=0.5, corr_seed=0.5,
                                corr_lo=0.0, corr_hi=1.0, march_axis=0,
                                rational=False, ctol=1e-9)
    assert not ok
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -k valley_floor_solve -v`
Expected: FAIL — `ImportError: cannot import name '_valley_floor_solve'`.

- [ ] **Step 3: Write minimal implementation**

```python
# mmcore/numeric/_bez_closest_point.py  (append in the valley section)

def _valley_floor_solve(S, point, march_val, corr_seed, corr_lo, corr_hi,
                        march_axis, rational, ctol, max_it=40):
    """Solve dg/d(corr) = 0 in [corr_lo, corr_hi] at fixed march coordinate.

    march_axis in {0,1}; the corrector axis is the other one. Returns
    (corr_value, ok). ok = strictly-interior root with positive cross-valley
    curvature (a genuine valley-floor point), not boundary-captured.
    """
    span = max(corr_hi - corr_lo, 1e-300)
    c = min(max(corr_seed, corr_lo), corr_hi)
    hit_boundary = False
    for _ in range(max_it):
        u, v = (march_val, c) if march_axis == 0 else (c, march_val)
        _, gu, gv, guu, _, gvv = _surface_g_derivs(S, point, u, v, rational)
        gc = gv if march_axis == 0 else gu      # dg/d(corr)
        gcc = gvv if march_axis == 0 else guu    # d2g/d(corr)^2
        if abs(gcc) < 1e-300:
            return c, False
        step = -gc / gcc
        cn = min(max(c + step, corr_lo), corr_hi)
        hit_boundary = (cn <= corr_lo + 1e-15) or (cn >= corr_hi - 1e-15)
        c = cn
        if abs(step) < ctol:
            break
    u, v = (march_val, c) if march_axis == 0 else (c, march_val)
    _, gu, gv, guu, _, gvv = _surface_g_derivs(S, point, u, v, rational)
    gcc = gvv if march_axis == 0 else guu
    eps = 1e-9 * span
    ok = (not hit_boundary) and (gcc > 0.0) and (corr_lo + eps < c < corr_hi - eps)
    return c, ok
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -k valley_floor_solve -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/_bez_closest_point.py tests/test_bez_closest_point.py
git commit -m "feat(closest-point): add valley-floor corrector"
```

---

### Task 4: The marcher `_march_valley_cell`

**Files:**
- Modify: `mmcore/numeric/_bez_closest_point.py`
- Test: `tests/test_bez_closest_point.py`

Sample the march axis across the cell, correct each sample onto the floor (continuation-seeded), detect `−→+` sign-changes of the along-valley gradient `φ` between contiguous floor samples, 2-D-Newton-polish + classify each into `out`. If `φ ≈ 0` over the whole floor, emit a single `degenerate_min` representative.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bez_closest_point.py  (append)
from mmcore.numeric._bez_closest_point import _march_valley_cell


def test_march_valley_cell_finds_two_minima():
    S = _ruled_U_surface(H=4.0)
    P = np.array([0.0, 1.0, 2.0])    # inside the U, mid-height: two valley minima
    out = []
    _march_valley_cell(S, P, out, 0.0, 1.0, 0.0, 1.0, march_axis=0,
                       rational=False, atol=1e-6, ptol_u=1e-4, ptol_v=1e-4)
    mins = [e for e in out if e["kind"] == "min"]
    assert len(mins) == 2
    # both minima at v=0.5 (P's height), on the two arms (u near the ends)
    for e in mins:
        assert abs(e["v"] - 0.5) < 1e-3
    us = sorted(e["u"] for e in mins)
    assert us[0] < 0.25 and us[1] > 0.75


def test_march_valley_cell_degenerate_manifold():
    # P exactly on the symmetric axis of a circular configuration: a flat valley.
    # Build a symmetric ruled surface (a straight segment extruded): every u gives
    # the same distance -> phi == 0 everywhere -> one degenerate_min.
    seg = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])      # degree-1 segment in x
    net = np.zeros((2, 2, 3))
    net[:, 0, :] = seg
    net[:, 1, :] = seg + np.array([0.0, 0.0, 4.0])
    P = np.array([0.0, 5.0, 2.0])    # equidistant from every (u, v=0.5): flat valley in u
    out = []
    _march_valley_cell(net, P, out, 0.0, 1.0, 0.0, 1.0, march_axis=0,
                       rational=False, atol=1e-6, ptol_u=1e-4, ptol_v=1e-4)
    assert len(out) == 1
    assert out[0]["kind"] == "degenerate_min"
    assert abs(out[0]["v"] - 0.5) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -k march_valley_cell -v`
Expected: FAIL — `ImportError: cannot import name '_march_valley_cell'`.

- [ ] **Step 3: Write minimal implementation**

```python
# mmcore/numeric/_bez_closest_point.py  (append in the valley section)

def _march_valley_cell(S, point, out, u0, u1, v0, v1, march_axis, rational,
                       atol, ptol_u, ptol_v, n_samples=48):
    """Trace the valley floor across a certified cell and add its minima to out.

    march_axis: 0 = march u / correct v ; 1 = march v / correct u (the corrector
    axis is the one whose cross-valley curvature was certified positive). Adds
    {"u","v","point","distance","kind"} entries via _dedup_add. Emits a single
    "degenerate_min" if the whole floor is equidistant.
    """
    m_lo, m_hi = (u0, u1) if march_axis == 0 else (v0, v1)
    c_lo, c_hi = (v0, v1) if march_axis == 0 else (u0, u1)
    ctol = 1e-12 * max(c_hi - c_lo, 1.0)
    ms = np.linspace(m_lo, m_hi, n_samples + 1)

    # Trace: floor[k] = (mval, c, phi, g) or None where the floor leaves the cell.
    floor = []
    c_seed = 0.5 * (c_lo + c_hi)
    for mval in ms:
        c, ok = _valley_floor_solve(S, point, mval, c_seed, c_lo, c_hi,
                                    march_axis, rational, ctol)
        if not ok:
            floor.append(None)
            continue
        c_seed = c
        u, v = (mval, c) if march_axis == 0 else (c, mval)
        _, gu, gv, _, _, _ = _surface_g_derivs(S, point, u, v, rational)
        phi = gu if march_axis == 0 else gv     # along-valley gradient = dh/d(march)
        g = _surface_g_value(S, point, u, v, rational)
        floor.append((mval, c, phi, g))

    valid = [f for f in floor if f is not None]
    if not valid:
        return

    # Degenerate minimum manifold: floor equidistant => phi ~ 0 throughout.
    gscale = max(abs(f[3]) for f in valid)
    if (max(abs(f[2]) for f in valid) <= 1e-7 * max(gscale, 1.0)) and len(valid) >= 3:
        mval, c, _, g = valid[len(valid) // 2]
        u, v = (mval, c) if march_axis == 0 else (c, mval)
        pt = eval_surface(S, u, v, rational=rational)
        _dedup_add(out, u, v, float(np.sqrt(max(g, 0.0))), np.asarray(pt),
                   "degenerate_min", ptol_u, ptol_v, atol)
        return

    # Event detection on CONTIGUOUS floor samples (skip across None gaps).
    for k in range(len(floor) - 1):
        a, b = floor[k], floor[k + 1]
        if a is None or b is None:
            continue
        if np.sign(a[2]) == np.sign(b[2]):
            continue
        if (b[2] - a[2]) <= 0.0:
            continue                       # +->- is a saddle/max, not a minimum
        # -> + crossing: linear-interpolate the march coordinate, re-correct, polish.
        mstar = a[0] - a[2] * (b[0] - a[0]) / (b[2] - a[2])
        cstar, ok = _valley_floor_solve(S, point, mstar, 0.5 * (a[1] + b[1]),
                                        c_lo, c_hi, march_axis, rational, ctol)
        ustar, vstar = (mstar, cstar) if march_axis == 0 else (cstar, mstar)
        uu, vv, _, _ = newton_surface_closest_point(
            S, point, ustar, vstar, rational=rational, bounds=(u0, u1, v0, v1))
        is_min, dist, pt = _classify_surface_min(S, point, uu, vv, rational, atol)
        if is_min:
            _dedup_add(out, uu, vv, dist, pt, "min", ptol_u, ptol_v, atol)


def _surface_g_value(S, point, u, v, rational):
    s = eval_surface(S, u, v, rational=rational) - point
    return float(np.dot(s, s))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -k march_valley_cell -v`
Expected: PASS (2 tests). If `test_march_valley_cell_finds_two_minima` finds only 1 due to too-coarse sampling, raise `n_samples` default to 64 (the only permitted tuning); do not weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/_bez_closest_point.py tests/test_bez_closest_point.py
git commit -m "feat(closest-point): add valley-floor marcher"
```

---

### Task 5: Detect valley cells and integrate the marcher into the loop

**Files:**
- Modify: `mmcore/numeric/_bez_closest_point.py` (`bez_surface_closest_points`)
- Test: `tests/test_bez_closest_point.py`

Carry the subdivided weight net `Sw` through the cell stack (needed for the rational sign-net certificate). At `depth >= D_VALLEY`, if exactly one of `g_uu`/`g_vv` is hull-positive over the cell, march along the *other* (shallow) axis and drop the cell.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bez_closest_point.py  (append)
import warnings as _warnings
from mmcore.numeric._bez_closest_point import nurbs_surface_closest_points


def test_surfcp1_no_blowup_and_correct():
    import examples.closest_point.surfcp1 as scp
    val, P = scp.val, scp.pt
    import time
    t = time.perf_counter()
    with _warnings.catch_warnings(record=True) as wl:
        _warnings.simplefilter("always")
        res = nurbs_surface_closest_points(val, P, atol=1e-3)
    dt = time.perf_counter() - t
    cap = [w for w in wl if "max_cells" in str(w.message)]
    assert not cap, "valley marcher should prevent the max_cells blow-up"
    assert dt < 5.0, f"expected fast result, took {dt:.1f}s"
    # Two symmetric global minima at distance ~8504.84 (validated ground truth).
    best = sorted(e["distance"] for e in res)[:2]
    assert abs(best[0] - 8504.837) < 1e-1
    assert abs(best[1] - 8504.837) < 1e-1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -k surfcp1_no_blowup -v`
Expected: FAIL — currently hits the `max_cells` cap (the `cap` assertion fails) and/or `dt >= 5`.

- [ ] **Step 3: Implement — carry the weight net and add valley detection**

Replace the interior-subdivision block in `bez_surface_closest_points` (from `out = []` through the `warnings.warn(...)` for the cap) with this version. The changes: a `D_VALLEY` constant; the stack tuple gains `Swc` (the cell weight net); a valley-detection branch before the small/leaf branch; weight-net subdivision in both split branches.

```python
    out = []

    # Depth at which a still-unpruned cell is treated as a candidate valley cell
    # (a transversal root localizes well before this; a persistent straddle is
    # the degenerate-valley signature).
    D_VALLEY = 6

    # Interior subdivision. Cells carry the subdivided weight net Swc so the
    # rational cross-valley curvature certificate can be built per cell.
    stack = [(F, Sw, Nu, Nv, 0.0, 1.0, 0.0, 1.0, 0)]
    cells = 0
    while stack and cells < max_cells:
        cells += 1
        Fc, Swc, Nuc, Nvc, u0, u1, v0, v1, depth = stack.pop()
        # Joint stationarity prune: a stationary point needs BOTH partials = 0.
        if _hull_excludes_zero(Nuc) or _hull_excludes_zero(Nvc):
            continue
        if (float(np.max(np.abs(Nuc))) <= flat_amp
                and float(np.max(np.abs(Nvc))) <= flat_amp):
            # Both partials ~ 0: flat/degenerate region, take one representative.
            um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            u, v, R, _ = newton_surface_closest_point(
                S, point, um, vm, rational=rational, bounds=(u0, u1, v0, v1))
            is_min, dist, pt = _classify_surface_min(S, point, u, v, rational, atol)
            if is_min:
                _dedup_add(out, u, v, dist, pt, "min", ptol_u, ptol_v, atol)
            continue

        # Valley detection: a persistent (deep) unpruned cell whose cross-valley
        # curvature is certified positive in EXACTLY ONE axis is a clean valley
        # cell -> trace the floor instead of subdividing it.
        if depth >= D_VALLEY:
            vv_pos = _hull_positive(_g_diag_sign_net(Nvc, Swc, 1))   # g_vv > 0 in cell
            uu_pos = _hull_positive(_g_diag_sign_net(Nuc, Swc, 0))   # g_uu > 0 in cell
            if vv_pos != uu_pos:
                # march the SHALLOW axis (the one whose curvature is NOT definite),
                # correct in the definite axis.
                march_axis = 0 if vv_pos else 1   # vv definite -> march u
                _march_valley_cell(S, point, out, u0, u1, v0, v1, march_axis,
                                   rational, atol, ptol_u, ptol_v)
                continue

        small = (u1 - u0) <= ptol_u and (v1 - v0) <= ptol_v
        if small or depth > 80:
            um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            u, v, R, _ = newton_surface_closest_point(
                S, point, um, vm, rational=rational, bounds=(u0, u1, v0, v1))
            is_min, dist, pt = _classify_surface_min(S, point, u, v, rational, atol)
            if is_min:
                _dedup_add(out, u, v, dist, pt, "min", ptol_u, ptol_v, atol)
            continue
        # Split the wider axis (carry F, Sw, Nu, Nv together).
        if (u1 - u0) >= (v1 - v0):
            um = 0.5 * (u0 + u1)
            FL, FR = _split_net(Fc, 0)
            SwL, SwR = _split_net(Swc, 0)
            NuL, NuR = _split_net(Nuc, 0)
            NvL, NvR = _split_net(Nvc, 0)
            stack.append((FL, SwL, NuL, NvL, u0, um, v0, v1, depth + 1))
            stack.append((FR, SwR, NuR, NvR, um, u1, v0, v1, depth + 1))
        else:
            vm = 0.5 * (v0 + v1)
            FL, FR = _split_net(Fc, 1)
            SwL, SwR = _split_net(Swc, 1)
            NuL, NuR = _split_net(Nuc, 1)
            NvL, NvR = _split_net(Nvc, 1)
            stack.append((FL, SwL, NuL, NvL, u0, u1, v0, vm, depth + 1))
            stack.append((FR, SwR, NuR, NvR, u0, u1, vm, v1, depth + 1))

    if cells >= max_cells and stack:
        warnings.warn(
            "bez_surface_closest_points: subdivision hit max_cells cap; "
            "result may be incomplete.")
```

- [ ] **Step 4: Run the new test AND the full module**

Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -k surfcp1_no_blowup -v`
Expected: PASS.
Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -q`
Expected: PASS (all prior tests unaffected — transversal cases never reach `D_VALLEY` with a one-sided definite certificate, so they don't march).

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/_bez_closest_point.py tests/test_bez_closest_point.py
git commit -m "feat(closest-point): detect and march degenerate valley cells"
```

---

### Task 6: Relative `ptol` floor safeguard

**Files:**
- Modify: `mmcore/numeric/_bez_closest_point.py` (the two `ptol` clamps in `bez_surface_closest_points` and the one in `bez_curve_closest_points`)
- Test: `tests/test_bez_closest_point.py`

A belt-and-suspenders termination guarantee: never chase parametric resolution below `_PTOL_FLOOR` of the (unit) patch domain. Engages only at pathological scales; for normal O(1) surfaces the computed `ptol` already exceeds it, so existing behavior is unchanged (final accuracy is preserved by the leaf Newton polish).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bez_closest_point.py  (append)
from mmcore.numeric import _bez_closest_point as _bzmod


def test_ptol_floor_constant_and_engagement():
    # The floor must exist and be a small relative value.
    assert hasattr(_bzmod, "_PTOL_FLOOR")
    assert 1e-9 < _bzmod._PTOL_FLOOR < 1e-2
    # A huge transversal (non-degenerate) surface must still terminate quickly:
    # a tilted plane scaled to 1e5 with the foot in the interior.
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1e5, 1e3]],
                  [[1e5, 0.0, 1e3], [1e5, 1e5, 0.0]]])
    P = np.array([3e4, 4e4, 5e4])
    import time
    t = time.perf_counter()
    with _warnings.catch_warnings(record=True) as wl:
        _warnings.simplefilter("always")
        res = _bzmod.bez_surface_closest_points(S, P, atol=1e-3, rational=False)
    dt = time.perf_counter() - t
    assert not [w for w in wl if "max_cells" in str(w.message)]
    assert dt < 3.0
    assert len(res) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -k ptol_floor -v`
Expected: FAIL — `_PTOL_FLOOR` is not defined (`AssertionError` on `hasattr`).

- [ ] **Step 3: Implement the floor**

Add the module constant near the top of `_bez_closest_point.py` (after the imports):

```python
# Relative floor on the parametric tolerance: subdivision never resolves below
# this fraction of the unit patch domain (final accuracy comes from the leaf
# Newton polish). Prevents pathological deep subdivision at large coordinate
# scales; engages only when the geometry-derived ptol is already smaller.
_PTOL_FLOOR = 1e-4
```

In `bez_surface_closest_points`, change the two clamps:

```python
    ptol_u = max(float(ptol_u), _PTOL_FLOOR)
    ptol_v = max(float(ptol_v), _PTOL_FLOOR)
```

In `bez_curve_closest_points`, change:

```python
    ptol = max(ptol, _PTOL_FLOOR)
```

(These replace the existing `max(..., 1e-12)` clamps.)

- [ ] **Step 4: Run the new test AND the full module**

Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -k ptol_floor -v`
Expected: PASS.
Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -q`
Expected: PASS (all — O(1)-scale tests have geometry-ptol above 1e-4, so the floor is a no-op there; the multi-span/seam/curve tests still pass).

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/_bez_closest_point.py tests/test_bez_closest_point.py
git commit -m "feat(closest-point): add relative ptol floor as a termination safeguard"
```

---

### Task 7: Analytic valley acceptance tests + non-regression

**Files:**
- Modify: `tests/test_bez_closest_point.py`

End-to-end coverage on analytic valleys with known minima, and confirmation the full suite is green.

- [ ] **Step 1: Write the tests**

```python
# tests/test_bez_closest_point.py  (append)
def test_ruled_U_valley_via_core_two_minima():
    # The ruled "U" surface (clean v-valley, two u-minima) through the public core.
    S = _ruled_U_surface(H=4.0)
    P = np.array([0.0, 1.0, 2.0])
    res = bez_surface_closest_points(S, P, atol=1e-6, rational=False)
    mins = [e for e in res if e["kind"] in ("min", "boundary_min")]
    # Dense-grid ground truth
    best = np.inf
    for u in np.linspace(0, 1, 200):
        for v in np.linspace(0, 1, 60):
            d = np.linalg.norm(eval_surface(S, u, v, rational=False) - P)
            best = min(best, d)
    assert abs(min(e["distance"] for e in res) - best) < 1e-2
    interior = [e for e in res if abs(e["v"] - 0.5) < 1e-2 and e["kind"] == "min"]
    assert len(interior) >= 2     # both valley minima recovered


def test_degenerate_axis_reports_manifold():
    # Straight segment extruded; P equidistant from the whole mid-height line.
    seg = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    S = np.zeros((2, 2, 3))
    S[:, 0, :] = seg
    S[:, 1, :] = seg + np.array([0.0, 0.0, 4.0])
    P = np.array([0.0, 5.0, 2.0])
    res = bez_surface_closest_points(S, P, atol=1e-6, rational=False)
    assert any(e["kind"] == "degenerate_min" for e in res)
```

- [ ] **Step 2: Run them (expect PASS — implementation already complete)**

Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -k "ruled_U_valley or degenerate_axis_reports" -v`
Expected: PASS. If the ruled-U interior minima come back as `boundary_min` (the two arms' minima sit near `u` extremes), relax `interior` to also accept `boundary_min` at `v≈0.5`; do not relax the distance assertion.

- [ ] **Step 3: Run the WHOLE module suite**

Run: `.venv/bin/python -m pytest tests/test_bez_closest_point.py -v`
Expected: PASS (all — the original 25 plus the new valley tests).

- [ ] **Step 4: Commit**

```bash
git add tests/test_bez_closest_point.py
git commit -m "test(closest-point): analytic valley + degenerate-manifold acceptance tests"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §3 hybrid: subdivision + certified valley-resolve | Task 5 |
| §4 certificate = cross-valley curvature hull-positive (`M_aa`) | Task 2 (+ used in 5) |
| §5 marcher: edge-seed, correct, `φ` events, 2-D Newton polish, classify | Tasks 3, 4 |
| §6 control-flow integration; per-cell certified, no cross-cell bookkeeping | Task 5 |
| §7 degenerate-manifold `degenerate_min` semantics | Task 4 (+ test in 7) |
| §8 completeness (certificate gates marching; folds fail → subdivide) | Task 5 (`vv_pos != uu_pos`; non-definite cells fall through to subdivide) |
| §9 relative `ptol` floor safeguard | Task 6 |
| §10 risks: trigger (`D_VALLEY`), pole-exit handoff (`ok=False` floor exit), cost (lazy: only at `depth>=D_VALLEY`) | Tasks 4, 5 |
| §11 testing: surfcp1, analytic valleys, degenerate manifold, non-regression | Tasks 5, 7 |
| §12 surface-only (no curve marcher) | respected — only `bez_surface_closest_points` changed for marching |

Decisions (§13): trigger via `D_VALLEY` + curvature-definite (1); per-cell marching (2); `ptol` floor included (3); explicit `degenerate_min` kind (4) — all as recommended.

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to". Every step has concrete code. The only permitted tunings (`n_samples`, accepting `boundary_min` for the ruled-U arms) are explicit fallbacks with a rationale, not placeholders.

**3. Type consistency:** `_surface_g_derivs` returns the 6-tuple `(g,gu,gv,guu,guv,gvv)` used identically in `_valley_floor_solve`, `_march_valley_cell`, and the certificate path. `_g_diag_sign_net(N_diag, Sw, axis)` and `_hull_positive(net)` signatures match their Task-5 call sites. `_march_valley_cell(S, point, out, u0, u1, v0, v1, march_axis, rational, atol, ptol_u, ptol_v, n_samples=48)` matches the Task-5 call (positional through `ptol_v`). The stack tuple `(F, Sw, Nu, Nv, u0, u1, v0, v1, depth)` is unpacked and repacked consistently in all three places (pop, both split branches). Result dicts keep the `{"u","v","point","distance","kind"}` shape; `kind` gains `"degenerate_min"` alongside `"min"`/`"boundary_min"`. `_PTOL_FLOOR` is referenced only after its Task-6 definition.

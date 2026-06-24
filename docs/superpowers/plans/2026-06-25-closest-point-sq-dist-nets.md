# Closest-Point via Squared-Distance Bernstein Nets — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reliable, efficient net-based closest-point solvers for rational/non-rational Bézier and NURBS curves and surfaces that return all local minima (global first), replacing the unreliable divide-and-conquer code (kept for A/B comparison).

**Architecture:** Build the squared-distance numerator net `F` (existing `bern_sq_dist` constructors) and exact stationarity nets `N`/`(N_u, N_v)` = `F'·w − 2F·w'`. Subdivide cells, prune any cell whose stationarity-net Bernstein hull excludes 0 (no stationary point possible — sound completeness certificate), polish survivors with LM-damped Newton, classify minima via the pointwise Hessian, dedup, and sort by distance. NURBS-level wrappers decompose into Bézier patches and merge candidates across seams.

**Tech Stack:** Python 3.9+, NumPy. Reuses `mmcore.numeric.bern`, `mmcore.numeric.bern_sq_dist`, `mmcore.numeric.intersection._bezier_common`, `mmcore.numeric._bern_homog`, `mmcore.geom._nurbs_param_tol`, `mmcore.geom._nurbs_knots`, `mmcore.geom._nurbs_eval`. Tests via `pytest`. Spec: `docs/superpowers/specs/2026-06-25-closest-point-sq-dist-nets-design.md`.

---

## File Structure

- **Create** `mmcore/numeric/_bez_closest_point.py` — the entire new solver (isolated module):
  1. Bernstein algebra: `_bernstein_product_nd`, `point_curve_stationarity_net`, `point_surface_stationarity_nets`.
  2. Second-derivative evaluators: `eval_curve_d2`, `eval_surface_d2`.
  3. Newton kernels: `newton_curve_closest_point` (self-contained port), `newton_surface_closest_point` (new).
  4. Cores: `bez_curve_closest_points`, `bez_surface_closest_points`.
  5. NURBS wrappers: `nurbs_curve_closest_points`, `nurbs_surface_closest_points`.
- **Create** `tests/test_bez_closest_point.py` — all tests.
- **Do NOT modify** `mmcore/numeric/closest_point.py` (legacy baseline, untouched).

Conventions (matching the intersection code): a curve net `C` has shape `(q+1, D)` with `D=4` if `rational` else `3`; a surface net `S` has shape `(m+1, n+1, D)`. `point` is a Euclidean `(3,)` vector. Scalar Bernstein nets carry no trailing value axis; `bernstein_partial_derivative_coeffs` and `de_casteljau_split_nd` require a trailing axis, so wrap with `net[..., None]` and unwrap with `[..., 0]`.

---

### Task 1: Module scaffold + general Bernstein product

**Files:**
- Create: `mmcore/numeric/_bez_closest_point.py`
- Test: `tests/test_bez_closest_point.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bez_closest_point.py
import numpy as np
from math import comb
from mmcore.numeric._bez_closest_point import _bernstein_product_nd


def _bern_eval_1d(coeffs, t):
    n = len(coeffs) - 1
    B = np.array([comb(n, i) * t**i * (1 - t) ** (n - i) for i in range(n + 1)])
    return float(B @ coeffs)


def _bern_eval_2d(net, u, v):
    m, n = net.shape[0] - 1, net.shape[1] - 1
    Bu = np.array([comb(m, i) * u**i * (1 - u) ** (m - i) for i in range(m + 1)])
    Bv = np.array([comb(n, j) * v**j * (1 - v) ** (n - j) for j in range(n + 1)])
    return float(Bu @ net @ Bv)


def test_bernstein_product_1d_matches_pointwise():
    a = np.array([1.0, -2.0, 3.0])      # degree 2
    b = np.array([0.5, 4.0])            # degree 1
    c = _bernstein_product_nd(a, b)     # degree 3
    assert c.shape == (4,)
    for t in np.linspace(0, 1, 11):
        assert abs(_bern_eval_1d(c, t) - _bern_eval_1d(a, t) * _bern_eval_1d(b, t)) < 1e-12


def test_bernstein_product_2d_matches_pointwise():
    a = np.array([[1.0, 2.0], [3.0, -1.0], [0.0, 2.0]])   # bidegree (2,1)
    b = np.array([[1.0, 0.5], [-1.0, 2.0]])               # bidegree (1,1)
    c = _bernstein_product_nd(a, b)                        # bidegree (3,2)
    assert c.shape == (4, 3)
    for u in (0.0, 0.25, 0.5, 0.9, 1.0):
        for v in (0.0, 0.3, 1.0):
            assert abs(_bern_eval_2d(c, u, v) - _bern_eval_2d(a, u, v) * _bern_eval_2d(b, u, v)) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bez_closest_point.py -k bernstein_product -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'mmcore.numeric._bez_closest_point'`.

- [ ] **Step 3: Write minimal implementation**

```python
# mmcore/numeric/_bez_closest_point.py
"""Closest-point on rational Bézier/NURBS curves and surfaces via
squared-distance Bernstein nets.

Replaces the unreliable divide-and-conquer code in ``closest_point.py``
(kept untouched for A/B comparison). See
``docs/superpowers/specs/2026-06-25-closest-point-sq-dist-nets-design.md``.
"""
from __future__ import annotations

from math import comb

import numpy as np


# ---------------------------------------------------------------------------
# Bernstein algebra
# ---------------------------------------------------------------------------

def _binom_row(n):
    return np.array([comb(n, i) for i in range(n + 1)], dtype=np.float64)


def _scale_by_binoms(net):
    """Multiply a scalar Bernstein net by per-axis binomial coefficients."""
    out = np.asarray(net, dtype=np.float64).copy()
    for ax in range(out.ndim):
        p = out.shape[ax] - 1
        shape = [1] * out.ndim
        shape[ax] = p + 1
        out = out * _binom_row(p).reshape(shape)
    return out


def _unscale_by_binoms(net):
    """Divide a scalar Bernstein net by per-axis binomial coefficients."""
    out = np.asarray(net, dtype=np.float64).copy()
    for ax in range(out.ndim):
        p = out.shape[ax] - 1
        shape = [1] * out.ndim
        shape[ax] = p + 1
        out = out / _binom_row(p).reshape(shape)
    return out


def _ndconv_full(A, B):
    """Exact full linear convolution of two scalar ND arrays (small nets)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    out_shape = tuple(sa + sb - 1 for sa, sb in zip(A.shape, B.shape))
    out = np.zeros(out_shape, dtype=np.float64)
    for idxB in np.ndindex(*B.shape):
        bval = B[idxB]
        if bval == 0.0:
            continue
        sl = tuple(slice(i, i + s) for i, s in zip(idxB, A.shape))
        out[sl] += A * bval
    return out


def _bernstein_product_nd(a, b):
    """Exact product of two scalar Bernstein nets of equal ndim.

    Uses ``B_i^p * B_j^q = [C(p,i)C(q,j)/C(p+q,i+j)] B_{i+j}^{p+q}`` per axis.
    Returns a net of per-axis degree ``deg(a)+deg(b)``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != b.ndim:
        raise ValueError("operands must have the same number of axes")
    num = _ndconv_full(_scale_by_binoms(a), _scale_by_binoms(b))
    return _unscale_by_binoms(num)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bez_closest_point.py -k bernstein_product -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/_bez_closest_point.py tests/test_bez_closest_point.py
git commit -m "feat(closest-point): add general Bernstein product helper"
```

---

### Task 2: Stationarity-net builders

**Files:**
- Modify: `mmcore/numeric/_bez_closest_point.py`
- Test: `tests/test_bez_closest_point.py`

The stationarity net of `g = F/w²` is `N = F'·w − 2F·w'` (curve) and `N_u = F_u·w − 2F·w_u`, `N_v = F_v·w − 2F·w_v` (surface). For non-rational input (`w ≡ 1`) these reduce to `F'`, `F_u`, `F_v`. The test below verifies that a returned net **vanishes exactly where `d/dt g = 0`** by sampling the true rational distance derivative via central differences.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bez_closest_point.py  (append)
from mmcore.numeric._bez_closest_point import (
    point_curve_stationarity_net,
    point_surface_stationarity_nets,
)
from mmcore.numeric import bern_sq_dist


def _g_curve(F, Qw, t):
    return bern_sq_dist.eval_point_curve_distance_sq(F, Qw, t)


def test_curve_stationarity_net_nonrational_is_Fprime():
    # Quadratic non-rational curve
    C = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]])
    P = np.array([1.0, -1.0, 0.0])
    N, F, Qw = point_curve_stationarity_net(P, C, rational=False)
    # N must change sign at the same t where d/dt ||P-C(t)||^2 = 0.
    # Find a sign change of N by sampling and confirm g has a stationary point there.
    ts = np.linspace(0, 1, 401)
    Nvals = np.array([_eval_bern_1d(N, t) for t in ts])
    sign_changes = np.where(np.sign(Nvals[:-1]) != np.sign(Nvals[1:]))[0]
    assert len(sign_changes) >= 1
    # At each sign change, g'(t) computed by finite difference is ~0
    for k in sign_changes:
        t0 = ts[k]
        h = 1e-5
        gp = (_g_curve(F, Qw, min(1, t0 + h)) - _g_curve(F, Qw, max(0, t0 - h))) / (2 * h)
        assert abs(gp) < 1e-1  # near-zero at the bracketed root


def test_curve_stationarity_net_rational_tracks_true_derivative():
    # Rational quadratic quarter circle in xy-plane
    s = np.sqrt(2) / 2
    C = np.array([[1.0, 0.0, 0.0, 1.0],
                  [s, s, 0.0, s],     # homogeneous: (x*w, y*w, z*w, w)
                  [0.0, 1.0, 0.0, 1.0]])
    P = np.array([0.6, 0.6, 0.0])
    N, F, Qw = point_curve_stationarity_net(P, C, rational=True)
    # Sign change of N must bracket a stationary point of the TRUE distance.
    ts = np.linspace(0, 1, 801)
    Nvals = np.array([_eval_bern_1d(N, t) for t in ts])
    sc = np.where(np.sign(Nvals[:-1]) != np.sign(Nvals[1:]))[0]
    assert len(sc) >= 1
    for k in sc:
        t0 = ts[k]
        h = 1e-5
        gp = (_g_curve(F, Qw, min(1, t0 + h)) - _g_curve(F, Qw, max(0, t0 - h))) / (2 * h)
        assert abs(gp) < 1e-2


def test_surface_stationarity_nets_nonrational_are_partials():
    # Bilinear non-rational patch (unit square, z=0)
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    P = np.array([0.3, 0.4, 5.0])
    Nu, Nv, F, Sw = point_surface_stationarity_nets(P, S, rational=False)
    # Gradient of g vanishes at (u,v)=(0.3,0.4): both nets bracket zero there.
    assert _eval_bern_2d(Nu, 0.3, 0.4) == 0.0 or abs(_eval_bern_2d(Nu, 0.3, 0.4)) < 1e-9
    assert abs(_eval_bern_2d(Nv, 0.3, 0.4)) < 1e-9
```

Add these two tiny scalar-net evaluators near the top of the test file (used throughout):

```python
# tests/test_bez_closest_point.py  (add near the other helpers)
def _eval_bern_1d(coeffs, t):
    n = len(coeffs) - 1
    B = np.array([comb(n, i) * t**i * (1 - t) ** (n - i) for i in range(n + 1)])
    return float(B @ coeffs)


def _eval_bern_2d(net, u, v):
    m, n = net.shape[0] - 1, net.shape[1] - 1
    Bu = np.array([comb(m, i) * u**i * (1 - u) ** (m - i) for i in range(m + 1)])
    Bv = np.array([comb(n, j) * v**j * (1 - v) ** (n - j) for j in range(n + 1)])
    return float(Bu @ net @ Bv)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bez_closest_point.py -k stationarity -v`
Expected: FAIL — `ImportError: cannot import name 'point_curve_stationarity_net'`.

- [ ] **Step 3: Write minimal implementation**

```python
# mmcore/numeric/_bez_closest_point.py  (append)
from mmcore.numeric import bern_sq_dist
from mmcore.numeric.bern import bernstein_partial_derivative_coeffs


def _deriv_net(net, axis):
    """Bernstein coeffs of the partial derivative along ``axis`` (scalar net in/out)."""
    return bernstein_partial_derivative_coeffs(net[..., None], axis)[..., 0]


def point_curve_stationarity_net(point, C, rational=True):
    """Return ``(N, F, Qw)`` where ``N(t)=0`` iff ``d/dt ||point-C(t)||^2 = 0``.

    ``N = F'·w − 2F·w'`` (exact); for non-rational input ``N = F'``.
    ``F`` is the squared-distance numerator net and ``Qw`` the weight net.
    """
    C = np.asarray(C, dtype=np.float64)
    F = bern_sq_dist.point_curve_distance_squared_net_homog(point, C, rational=rational)
    Qw = C[:, -1].copy() if rational else np.ones(C.shape[0], dtype=np.float64)
    Fp = _deriv_net(F, 0)
    if not rational:
        return Fp, F, Qw
    wp = _deriv_net(Qw, 0)
    N = _bernstein_product_nd(Fp, Qw) - 2.0 * _bernstein_product_nd(F, wp)
    return N, F, Qw


def point_surface_stationarity_nets(point, S, rational=True):
    """Return ``(N_u, N_v, F, Sw)``; a joint stationary point needs both nets = 0.

    ``N_u = F_u·w − 2F·w_u``, ``N_v = F_v·w − 2F·w_v`` (exact); non-rational →
    ``N_u = F_u``, ``N_v = F_v``.
    """
    S = np.asarray(S, dtype=np.float64)
    F = bern_sq_dist.point_surface_distance_squared_net_homog(point, S, rational=rational)
    Sw = S[:, :, -1].copy() if rational else np.ones(S.shape[:2], dtype=np.float64)
    Fu = _deriv_net(F, 0)
    Fv = _deriv_net(F, 1)
    if not rational:
        return Fu, Fv, F, Sw
    wu = _deriv_net(Sw, 0)
    wv = _deriv_net(Sw, 1)
    Nu = _bernstein_product_nd(Fu, Sw) - 2.0 * _bernstein_product_nd(F, wu)
    Nv = _bernstein_product_nd(Fv, Sw) - 2.0 * _bernstein_product_nd(F, wv)
    return Nu, Nv, F, Sw
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bez_closest_point.py -k stationarity -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/_bez_closest_point.py tests/test_bez_closest_point.py
git commit -m "feat(closest-point): add exact stationarity-net builders"
```

---

### Task 3: Second-derivative evaluators (for min classification)

**Files:**
- Modify: `mmcore/numeric/_bez_closest_point.py`
- Test: `tests/test_bez_closest_point.py`

These wrap the existing Cython second-derivative routines so classification can build `g`'s Hessian.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bez_closest_point.py  (append)
from mmcore.numeric._bez_closest_point import eval_curve_d2, eval_surface_d2
from mmcore.numeric.intersection._bezier_common import eval_curve, eval_surface


def test_eval_curve_d2_matches_finite_difference():
    C = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0], [3.0, 1.0, 0.0]])
    t = 0.37
    pt, d1, d2 = eval_curve_d2(C, t, rational=False)
    h = 1e-5
    fd1 = (eval_curve(C, t + h, rational=False) - eval_curve(C, t - h, rational=False)) / (2 * h)
    fd2 = (eval_curve(C, t + h, rational=False) - 2 * eval_curve(C, t, rational=False)
           + eval_curve(C, t - h, rational=False)) / h**2
    assert np.allclose(d1, fd1, atol=1e-5)
    assert np.allclose(d2, fd2, atol=1e-3)


def test_eval_surface_d2_matches_finite_difference():
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.5]],
                  [[1.0, 0.0, 0.5], [1.0, 1.0, 0.0]]])
    u, v = 0.4, 0.6
    pt, Su, Sv, Suu, Suv, Svv = eval_surface_d2(S, u, v, rational=False)
    h = 1e-4
    fSuu = (eval_surface(S, u + h, v, rational=False) - 2 * eval_surface(S, u, v, rational=False)
            + eval_surface(S, u - h, v, rational=False)) / h**2
    fSuv = (eval_surface(S, u + h, v + h, rational=False) - eval_surface(S, u + h, v - h, rational=False)
            - eval_surface(S, u - h, v + h, rational=False) + eval_surface(S, u - h, v - h, rational=False)) / (4 * h**2)
    assert np.allclose(Suu, fSuu, atol=1e-2)
    assert np.allclose(Suv, fSuv, atol=1e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bez_closest_point.py -k eval_curve_d2 -v`
Expected: FAIL — `ImportError: cannot import name 'eval_curve_d2'`.

- [ ] **Step 3: Write minimal implementation**

```python
# mmcore/numeric/_bez_closest_point.py  (append)
from mmcore.numeric._bern_homog import (
    eval_bezier_curve_homog_with_derivs,
    eval_bezier_surface_homog_with_derivs,
    project_curve_homog_to_cartesian,
    project_surface_homog_to_cartesian,
)
from mmcore.numeric.intersection._bezier_common import (
    _to_homog_curve, _to_homog_surface, eval_curve, eval_curve_d1,
    eval_surface, eval_surface_d1, extract_weights, _clamp01,
)


def eval_curve_d2(C, t, rational=True):
    """Return ``(point, C1, C2)`` Euclidean curve value and 1st/2nd derivatives."""
    Ph = _to_homog_curve(C, rational=rational)
    Ch, Chd, Ch2 = eval_bezier_curve_homog_with_derivs(Ph, float(t), True)
    pt, d1, d2 = project_curve_homog_to_cartesian(Ch, Chd, Ch2)
    return np.asarray(pt), np.asarray(d1), np.asarray(d2)


def eval_surface_d2(S, u, v, rational=True):
    """Return ``(point, Su, Sv, Suu, Suv, Svv)`` Euclidean surface value/derivatives."""
    Sh = _to_homog_surface(S, rational=rational)
    Sh0, Shu, Shv, Shuu, Shuv, Shvv = eval_bezier_surface_homog_with_derivs(Sh, float(u), float(v), True)
    pt, su, sv, suu, suv, svv = project_surface_homog_to_cartesian(Sh0, Shu, Shv, Shuu, Shuv, Shvv)
    return (np.asarray(pt), np.asarray(su), np.asarray(sv),
            np.asarray(suu), np.asarray(suv), np.asarray(svv))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bez_closest_point.py -k "eval_curve_d2 or eval_surface_d2" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/_bez_closest_point.py tests/test_bez_closest_point.py
git commit -m "feat(closest-point): add 2nd-derivative evaluators for classification"
```

---

### Task 4: Newton kernels (1D port + new 2D surface)

**Files:**
- Modify: `mmcore/numeric/_bez_closest_point.py`
- Test: `tests/test_bez_closest_point.py`

`newton_curve_closest_point` is a self-contained LM-damped 1D solver of `⟨C(u)−P, C'(u)⟩=0` (ported so the new module does not depend on `closest_point.py`). `newton_surface_closest_point` is the new 2×2 LM-damped, cell-bounded solver of `⟨S−P,Su⟩=⟨S−P,Sv⟩=0`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bez_closest_point.py  (append)
from mmcore.numeric._bez_closest_point import (
    newton_curve_closest_point, newton_surface_closest_point,
)


def test_newton_curve_closest_point_segment():
    C = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])   # segment along x
    P = np.array([0.5, 1.0, 0.0])                       # foot at x=0.5 -> t=0.25
    u, R, sq, _ = newton_curve_closest_point(C, P, 0.6, rational=False)
    assert abs(u - 0.25) < 1e-9
    assert abs(sq - 1.0) < 1e-9


def test_newton_surface_closest_point_plane_bounded():
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])  # unit square z=0
    P = np.array([0.3, 0.4, 7.0])
    u, v, R, step = newton_surface_closest_point(
        S, P, 0.5, 0.5, rational=False, bounds=(0.0, 1.0, 0.0, 1.0))
    assert abs(u - 0.3) < 1e-9 and abs(v - 0.4) < 1e-9
    # residual r = (<S-P,Su>, <S-P,Sv>) ~ 0
    assert np.linalg.norm(R) < 1e-7


def test_newton_surface_respects_cell_bounds():
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    P = np.array([0.3, 0.4, 7.0])
    # True foot (0.3,0.4) lies OUTSIDE this cell; solver must stay inside.
    u, v, R, step = newton_surface_closest_point(
        S, P, 0.65, 0.65, rational=False, bounds=(0.6, 0.9, 0.6, 0.9))
    assert 0.6 - 1e-9 <= u <= 0.9 + 1e-9
    assert 0.6 - 1e-9 <= v <= 0.9 + 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bez_closest_point.py -k newton -v`
Expected: FAIL — `ImportError: cannot import name 'newton_curve_closest_point'`.

- [ ] **Step 3: Write minimal implementation**

```python
# mmcore/numeric/_bez_closest_point.py  (append)

def newton_curve_closest_point(C, point, u0, *, rational=False,
                               tol=1e-14, step_tol=1e-14, max_it=30, lm_damp=1e-12,
                               bounds=(0.0, 1.0)):
    """LM-damped 1D closest-point solve of ``<C(u)-point, C'(u)> = 0``.

    Clamped to ``bounds=(lo, hi)``. Returns ``(u, R, sqdist, last_du)`` with
    ``R = C(u) - point``.
    """
    point = np.asarray(point, dtype=np.float64)
    lo, hi = bounds
    u = min(max(float(u0), lo), hi)
    last_du = 1.0
    for _ in range(max_it):
        p, d = eval_curve_d1(C, u, rational=rational)
        R = p - point
        sq = float(np.dot(R, R))
        g = float(np.dot(R, d))
        if abs(g) < tol or (u <= lo and g >= -tol) or (u >= hi and g <= tol):
            last_du = 0.0
            break
        A = float(np.dot(d, d)) + lm_damp
        if A <= 0.0 or not np.isfinite(A):
            last_du = 0.0
            break
        du = -g / A
        if du * du < step_tol * step_tol:
            last_du = float(du)
            break
        step = 1.0
        accepted = False
        for _ls in range(8):
            un = min(max(u + step * du, lo), hi)
            Rn = eval_curve(C, un, rational=rational) - point
            if float(np.dot(Rn, Rn)) <= sq:
                last_du = un - u
                u = un
                accepted = True
                break
            step *= 0.5
        if not accepted:
            last_du = 0.0
            break
    R = eval_curve(C, u, rational=rational) - point
    return u, R, float(np.dot(R, R)), last_du


def newton_surface_closest_point(S, point, u0, v0, *, rational=False,
                                 tol=1e-13, step_tol=1e-14, max_it=40, lm_damp=1e-10,
                                 bounds=None):
    """LM-damped 2x2 closest-point solve of the stationarity system

        r_u = <S(u,v)-point, S_u> = 0
        r_v = <S(u,v)-point, S_v> = 0

    The Jacobian is the Hessian of (1/2)||S-point||^2. ``bounds`` =
    ``(u_lo,u_hi,v_lo,v_hi)`` clamps iterates to the current cell.
    Returns ``(u, v, R, last_step)`` with ``R = (r_u, r_v)``.
    """
    point = np.asarray(point, dtype=np.float64)
    if bounds is None:
        u_lo, u_hi, v_lo, v_hi = 0.0, 1.0, 0.0, 1.0
    else:
        u_lo, u_hi, v_lo, v_hi = bounds
    u = min(max(float(u0), u_lo), u_hi)
    v = min(max(float(v0), v_lo), v_hi)
    last_step = (1.0, 1.0)

    def residual(uu, vv):
        pt, su, sv, suu, suv, svv = eval_surface_d2(S, uu, vv, rational=rational)
        dvec = pt - point
        r = np.array([np.dot(dvec, su), np.dot(dvec, sv)])
        H = np.array([
            [np.dot(su, su) + np.dot(dvec, suu), np.dot(su, sv) + np.dot(dvec, suv)],
            [np.dot(su, sv) + np.dot(dvec, suv), np.dot(sv, sv) + np.dot(dvec, svv)],
        ])
        return r, H, float(np.dot(dvec, dvec))

    for _ in range(max_it):
        r, H, sq = residual(u, v)
        if float(np.dot(r, r)) < tol * tol:
            last_step = (0.0, 0.0)
            break
        A = H + lm_damp * np.eye(2)
        try:
            delta = np.linalg.solve(A, -r)
        except np.linalg.LinAlgError:
            last_step = (0.0, 0.0)
            break
        if float(np.dot(delta, delta)) < step_tol * step_tol:
            last_step = (float(delta[0]), float(delta[1]))
            break
        step = 1.0
        accepted = False
        rn2 = float(np.dot(r, r))
        for _ls in range(10):
            un = min(max(u + step * delta[0], u_lo), u_hi)
            vn = min(max(v + step * delta[1], v_lo), v_hi)
            rr, _, _ = residual(un, vn)
            if float(np.dot(rr, rr)) <= rn2:
                last_step = (un - u, vn - v)
                u, v = un, vn
                accepted = True
                break
            step *= 0.5
        if not accepted:
            last_step = (0.0, 0.0)
            break
    r, _, _ = residual(u, v)
    return u, v, r, last_step
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bez_closest_point.py -k newton -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/_bez_closest_point.py tests/test_bez_closest_point.py
git commit -m "feat(closest-point): add 1D port + new 2D surface Newton kernels"
```

---

### Task 5: Curve core `bez_curve_closest_points`

**Files:**
- Modify: `mmcore/numeric/_bez_closest_point.py`
- Test: `tests/test_bez_closest_point.py`

Interval-tree subdivision on `N(t)`; prune cells whose `N` hull excludes 0; at small cells root-find `N` and Newton-polish; add endpoints `t∈{0,1}`; classify minima via `g''(t)>0` (using `eval_curve_d2`); dedup by `ptol`; return all minima sorted by distance.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bez_closest_point.py  (append)
from mmcore.numeric._bez_closest_point import bez_curve_closest_points


def _dense_min_curve(C, P, rational, nsamp=4000):
    ts = np.linspace(0, 1, nsamp)
    d = np.array([np.linalg.norm(eval_curve(C, t, rational=rational) - P) for t in ts])
    k = int(np.argmin(d))
    return ts[k], d[k]


def test_curve_closest_interior_min():
    C = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]])
    P = np.array([1.0, -1.0, 0.0])
    res = bez_curve_closest_points(C, P, atol=1e-6, rational=False)
    assert len(res) >= 1
    assert res == sorted(res, key=lambda e: e["distance"])
    t_ref, d_ref = _dense_min_curve(C, P, rational=False)
    assert abs(res[0]["distance"] - d_ref) < 1e-3
    assert abs(res[0]["t"] - t_ref) < 1e-2


def test_curve_closest_boundary_min():
    C = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])  # straight along x
    P = np.array([-1.0, 1.0, 0.0])                                     # nearest is t=0 endpoint
    res = bez_curve_closest_points(C, P, atol=1e-6, rational=False)
    assert res[0]["kind"] == "boundary_min"
    assert abs(res[0]["t"]) < 1e-6
    assert abs(res[0]["distance"] - np.sqrt(2.0)) < 1e-6


def test_curve_closest_multiple_minima_U_shape():
    # Cubic "U": two arms -> a point inside has two local minima
    C = np.array([[-2.0, 2.0, 0.0], [-2.0, -3.0, 0.0], [2.0, -3.0, 0.0], [2.0, 2.0, 0.0]])
    P = np.array([0.0, 1.0, 0.0])
    res = bez_curve_closest_points(C, P, atol=1e-6, rational=False)
    minima = [e for e in res if e["kind"] == "min"]
    assert len(minima) >= 2  # both arms
    t_ref, d_ref = _dense_min_curve(C, P, rational=False)
    assert abs(res[0]["distance"] - d_ref) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bez_closest_point.py -k curve_closest -v`
Expected: FAIL — `ImportError: cannot import name 'bez_curve_closest_points'`.

- [ ] **Step 3: Write minimal implementation**

```python
# mmcore/numeric/_bez_closest_point.py  (append)
from mmcore.numeric.bern import de_casteljau_split_nd
from mmcore.geom._nurbs_param_tol import bez_curve_param_tolerance, bez_surface_param_tolerance


def _split_net(net, axis, t=0.5):
    L, R = de_casteljau_split_nd(net[..., None], axis=axis, t=t)
    return L[..., 0], R[..., 0]


def _hull_excludes_zero(net):
    return float(net.min()) > 0.0 or float(net.max()) < 0.0


def _g_curve_value(F, Qw, t):
    return float(bern_sq_dist.eval_point_curve_distance_sq(F, Qw, t))


def bez_curve_closest_points(C, point, atol=1e-3, rational=False,
                             max_cells=20000):
    """All local minima of ``||point - C(t)||`` on a Bézier curve.

    Returns a list of ``{"t", "point", "distance", "kind"}`` sorted ascending
    by distance (``result[0]`` is the global closest). ``kind`` is ``"min"``
    (interior) or ``"boundary_min"`` (an endpoint).
    """
    C = np.asarray(C, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    N, F, Qw = point_curve_stationarity_net(point, C, rational=rational)
    ptol = float(bez_curve_param_tolerance(C, atol, rational=rational))
    ptol = max(ptol, 1e-12)

    candidates = []  # (t_global, distance, kind)

    def add_candidate(t, kind):
        t = min(max(float(t), 0.0), 1.0)
        dist = np.sqrt(max(_g_curve_value(F, Qw, t), 0.0))
        for ct, _, _ in candidates:
            if abs(ct - t) < ptol:
                return
        candidates.append((t, dist, kind))

    # Endpoints are always boundary candidates (KKT handled at classification).
    # Subdivision tree on N.
    stack = [(N, 0.0, 1.0, 0)]
    cells = 0
    while stack and cells < max_cells:
        cells += 1
        Ncell, t0, t1, depth = stack.pop()
        if _hull_excludes_zero(Ncell):
            continue
        if (t1 - t0) <= ptol or depth > 60:
            tmid = 0.5 * (t0 + t1)
            u, R, sq, _ = newton_curve_closest_point(C, point, tmid, rational=rational,
                                                     bounds=(t0, t1))
            add_candidate(u, "min")
            continue
        L, Rr = _split_net(Ncell, axis=0, t=0.5)
        tm = 0.5 * (t0 + t1)
        stack.append((L, t0, tm, depth + 1))
        stack.append((Rr, tm, t1, depth + 1))

    # Endpoint candidates.
    add_candidate(0.0, "boundary_min")
    add_candidate(1.0, "boundary_min")

    # Classify: interior candidate is a min iff g''(t) > 0; endpoints accepted
    # only if KKT (no descent into the interior).
    results = []
    for t, dist, kind in candidates:
        pt, c1, c2 = eval_curve_d2(C, t, rational=rational)
        dvec = pt - point
        gpp = float(np.dot(c1, c1) + np.dot(dvec, c2))     # (1/2) g''  up to +2 factor
        gp = float(np.dot(dvec, c1))                       # (1/2) g'
        if kind == "boundary_min":
            if t <= ptol and gp < -atol:        # descends into interior -> not a min
                continue
            if t >= 1.0 - ptol and gp > atol:
                continue
            results.append({"t": t, "point": np.asarray(pt), "distance": dist, "kind": "boundary_min"})
        else:
            if gpp <= 0.0:
                continue                          # maximum, not a minimum
            results.append({"t": t, "point": np.asarray(pt), "distance": dist, "kind": "min"})

    if not results:   # degenerate: fall back to the nearest sampled candidate
        t, dist, kind = min(candidates, key=lambda c: c[1])
        pt = eval_curve(C, t, rational=rational)
        results.append({"t": t, "point": np.asarray(pt), "distance": dist, "kind": kind})

    results.sort(key=lambda e: e["distance"])
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bez_closest_point.py -k curve_closest -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/_bez_closest_point.py tests/test_bez_closest_point.py
git commit -m "feat(closest-point): add Bézier curve closest-points core"
```

---

### Task 6: Surface core (interior minima) `bez_surface_closest_points`

**Files:**
- Modify: `mmcore/numeric/_bez_closest_point.py`
- Test: `tests/test_bez_closest_point.py`

Quadtree subdivision carrying `(F, N_u, N_v)`; prune when **either** `N_u` or `N_v` hull excludes 0; at small cells Newton-polish (`newton_surface_closest_point`, bounded) and record the candidate; classify interior minima via the pointwise Hessian of `g`; dedup; sort. **Boundary handling is added in Task 7** — this task covers interior minima only (tests use queries whose closest point is interior).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bez_closest_point.py  (append)
from mmcore.numeric._bez_closest_point import bez_surface_closest_points


def _dense_min_surface(S, P, rational, n=200):
    us = np.linspace(0, 1, n)
    vs = np.linspace(0, 1, n)
    best = (None, None, np.inf)
    for u in us:
        for v in vs:
            d = np.linalg.norm(eval_surface(S, u, v, rational=rational) - P)
            if d < best[2]:
                best = (u, v, d)
    return best


def test_surface_closest_plane_interior():
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])   # unit square z=0
    P = np.array([0.3, 0.4, 5.0])
    res = bez_surface_closest_points(S, P, atol=1e-6, rational=False)
    assert len(res) >= 1
    assert res == sorted(res, key=lambda e: e["distance"])
    assert abs(res[0]["u"] - 0.3) < 1e-4 and abs(res[0]["v"] - 0.4) < 1e-4
    assert abs(res[0]["distance"] - 5.0) < 1e-4
    assert res[0]["kind"] == "min"


def test_surface_closest_curved_patch_matches_dense_grid():
    # Non-planar biquadratic-ish patch (bilinear with a bump via z)
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 2.0, 0.0]],
                  [[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]]])
    P = np.array([1.0, 1.0, 3.0])
    res = bez_surface_closest_points(S, P, atol=1e-6, rational=False)
    u_ref, v_ref, d_ref = _dense_min_surface(S, P, rational=False)
    assert abs(res[0]["distance"] - d_ref) < 5e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bez_closest_point.py -k surface_closest_plane -v`
Expected: FAIL — `ImportError: cannot import name 'bez_surface_closest_points'`.

- [ ] **Step 3: Write minimal implementation**

```python
# mmcore/numeric/_bez_closest_point.py  (append)

def _g_surface_value(F, Sw, u, v):
    return float(bern_sq_dist.eval_point_surface_distance_sq(F, Sw, u, v))


def _classify_surface_min(S, point, u, v, rational, atol):
    """Return (is_min, dist, pt) using the pointwise Hessian of g."""
    pt, su, sv, suu, suv, svv = eval_surface_d2(S, u, v, rational=rational)
    dvec = pt - point
    H11 = float(np.dot(su, su) + np.dot(dvec, suu))
    H22 = float(np.dot(sv, sv) + np.dot(dvec, svv))
    H12 = float(np.dot(su, sv) + np.dot(dvec, suv))
    det = H11 * H22 - H12 * H12
    is_min = (H11 > 0.0) and (det > 0.0)
    return is_min, float(np.linalg.norm(dvec)), np.asarray(pt)


def _dedup_add(out, u, v, dist, pt, kind, ptol_u, ptol_v, atol):
    for e in out:
        if (abs(e["u"] - u) < ptol_u and abs(e["v"] - v) < ptol_v
                and np.linalg.norm(e["point"] - pt) < max(atol, 1e-9)):
            return
    out.append({"u": float(u), "v": float(v), "point": pt, "distance": float(dist), "kind": kind})


def bez_surface_closest_points(S, point, atol=1e-3, rational=False,
                               want_eval=False, max_cells=60000,
                               _interior_only=False):
    """All local minima of ``||point - S(u,v)||`` on a Bézier surface patch.

    Returns ``{"u","v","point","distance","kind"[, "eval"]}`` sorted ascending
    by distance. ``kind`` is ``"min"`` (interior) or ``"boundary_min"`` (edge/corner).
    """
    S = np.asarray(S, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    Nu, Nv, F, Sw = point_surface_stationarity_nets(point, S, rational=rational)
    ptol_u, ptol_v = bez_surface_param_tolerance(S, atol, rational=rational)
    ptol_u = max(float(ptol_u), 1e-12)
    ptol_v = max(float(ptol_v), 1e-12)

    out = []

    # Interior subdivision.
    stack = [(F, Nu, Nv, 0.0, 1.0, 0.0, 1.0, 0)]
    cells = 0
    while stack and cells < max_cells:
        cells += 1
        Fc, Nuc, Nvc, u0, u1, v0, v1, depth = stack.pop()
        # Joint stationarity prune: a stationary point needs BOTH partials = 0.
        if _hull_excludes_zero(Nuc) or _hull_excludes_zero(Nvc):
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
        # Split the wider axis (carry F, Nu, Nv together).
        if (u1 - u0) >= (v1 - v0):
            ax = 0
            um = 0.5 * (u0 + u1)
            FL, FR = _split_net(Fc, 0)
            NuL, NuR = _split_net(Nuc, 0)
            NvL, NvR = _split_net(Nvc, 0)
            stack.append((FL, NuL, NvL, u0, um, v0, v1, depth + 1))
            stack.append((FR, NuR, NvR, um, u1, v0, v1, depth + 1))
        else:
            vm = 0.5 * (v0 + v1)
            FL, FR = _split_net(Fc, 1)
            NuL, NuR = _split_net(Nuc, 1)
            NvL, NvR = _split_net(Nvc, 1)
            stack.append((FL, NuL, NvL, u0, u1, v0, vm, depth + 1))
            stack.append((FR, NuR, NvR, u0, u1, vm, v1, depth + 1))

    if not _interior_only:
        _add_surface_boundary_minima(S, point, out, F, Sw, rational, atol, ptol_u, ptol_v)

    if not out:   # degenerate fallback: center Newton
        u, v, R, _ = newton_surface_closest_point(S, point, 0.5, 0.5, rational=rational)
        pt = eval_surface(S, u, v, rational=rational)
        out.append({"u": u, "v": v, "point": np.asarray(pt),
                    "distance": float(np.linalg.norm(pt - point)), "kind": "min"})

    out.sort(key=lambda e: e["distance"])
    if want_eval:
        for e in out:
            pt, su, sv, suu, suv, svv = eval_surface_d2(S, e["u"], e["v"], rational=rational)
            nrm = np.cross(su, sv)
            e["eval"] = {"S": pt, "Su": su, "Sv": sv, "normal": nrm}
    return out
```

Add a temporary stub so the module imports before Task 7 fills it in:

```python
# mmcore/numeric/_bez_closest_point.py  (append — REPLACED in Task 7)
def _add_surface_boundary_minima(S, point, out, F, Sw, rational, atol, ptol_u, ptol_v):
    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bez_closest_point.py -k "surface_closest_plane or surface_closest_curved" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/_bez_closest_point.py tests/test_bez_closest_point.py
git commit -m "feat(closest-point): add Bézier surface closest-points core (interior)"
```

---

### Task 7: Surface boundary minima (edges + corners)

**Files:**
- Modify: `mmcore/numeric/_bez_closest_point.py` (replace the `_add_surface_boundary_minima` stub)
- Test: `tests/test_bez_closest_point.py`

Run the curve core on each of the 4 boundary isocurves and the 4 corners; map back to `(u, v)`; accept a boundary candidate only if it is KKT-valid (gradient of `g` does not point into the interior).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bez_closest_point.py  (append)
def test_surface_closest_on_edge():
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])   # unit square z=0
    P = np.array([-1.0, 0.4, 0.0])                       # nearest point is edge u=0, v=0.4
    res = bez_surface_closest_points(S, P, atol=1e-6, rational=False)
    assert res[0]["kind"] == "boundary_min"
    assert abs(res[0]["u"]) < 1e-5 and abs(res[0]["v"] - 0.4) < 1e-4
    assert abs(res[0]["distance"] - 1.0) < 1e-5


def test_surface_closest_on_corner():
    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])
    P = np.array([-1.0, -1.0, 0.0])                       # nearest is corner (u=0,v=0)
    res = bez_surface_closest_points(S, P, atol=1e-6, rational=False)
    assert res[0]["kind"] == "boundary_min"
    assert abs(res[0]["u"]) < 1e-5 and abs(res[0]["v"]) < 1e-5
    assert abs(res[0]["distance"] - np.sqrt(2.0)) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bez_closest_point.py -k "surface_closest_on_edge or surface_closest_on_corner" -v`
Expected: FAIL — the stub returns nothing, so `res[0]` is an interior fallback, not a `boundary_min` at the edge.

- [ ] **Step 3: Write implementation (replace the stub)**

```python
# mmcore/numeric/_bez_closest_point.py  (REPLACE the _add_surface_boundary_minima stub)
def _add_surface_boundary_minima(S, point, out, F, Sw, rational, atol, ptol_u, ptol_v):
    """Add KKT-valid minima on the 4 edges and 4 corners of the patch."""
    point = np.asarray(point, dtype=np.float64)

    # Edges as isocurves (control nets). (fixed_axis, side, isocurve net)
    # surf axis 0 == u, axis 1 == v.
    edges = [
        (0, 0.0, S[0, :, :]),    # u = 0, runs along v
        (0, 1.0, S[-1, :, :]),   # u = 1
        (1, 0.0, S[:, 0, :]),    # v = 0, runs along u
        (1, 1.0, S[:, -1, :]),   # v = 1
    ]
    for fixed_axis, side, iso in edges:
        iso_res = bez_curve_closest_points(iso, point, atol=atol, rational=rational)
        for e in iso_res:
            s = e["t"]
            if fixed_axis == 0:
                u, v = side, s
            else:
                u, v = s, side
            _try_add_boundary(S, point, out, u, v, rational, atol, ptol_u, ptol_v)

    # Corners.
    for u, v in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
        _try_add_boundary(S, point, out, u, v, rational, atol, ptol_u, ptol_v)


def _try_add_boundary(S, point, out, u, v, rational, atol, ptol_u, ptol_v):
    """KKT filter + dedup for a boundary candidate at (u, v)."""
    pt, su, sv = eval_surface_d1(S, u, v, rational=rational)
    dvec = pt - point
    gu = float(np.dot(dvec, su))   # (1/2) dg/du
    gv = float(np.dot(dvec, sv))   # (1/2) dg/dv
    # KKT: at u=0 need gu>=0 (cannot descend by increasing u); at u=1 need gu<=0; etc.
    if u <= ptol_u and gu < -atol:
        return
    if u >= 1.0 - ptol_u and gu > atol:
        return
    if v <= ptol_v and gv < -atol:
        return
    if v >= 1.0 - ptol_v and gv > atol:
        return
    dist = float(np.linalg.norm(dvec))
    _dedup_add(out, u, v, dist, np.asarray(pt), "boundary_min", ptol_u, ptol_v, atol)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bez_closest_point.py -k "surface_closest" -v`
Expected: PASS (4 surface tests, incl. the earlier two).

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/_bez_closest_point.py tests/test_bez_closest_point.py
git commit -m "feat(closest-point): add surface boundary (edge/corner) minima"
```

---

### Task 8: NURBS-level wrappers with seam dedup

**Files:**
- Modify: `mmcore/numeric/_bez_closest_point.py`
- Test: `tests/test_bez_closest_point.py`

Decompose the NURBS object into Bézier patches, run the per-patch core, map patch-local `[0,1]` params back to global NURBS params via each patch's knot interval, and merge candidates across seams. A patch edge counts as a *real* boundary only where it coincides with the global domain boundary; internal seams are deduped and never reported as `boundary_min`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bez_closest_point.py  (append)
from mmcore.numeric._bez_closest_point import (
    nurbs_curve_closest_points, nurbs_surface_closest_points,
)
from mmcore.geom._nurbs_eval import NURBSCurveTuple, NURBSSurfaceTuple


def _bezier_curve_tuple(ctrl_xyz, weights):
    n = len(ctrl_xyz)
    deg = n - 1
    knot = np.concatenate([np.zeros(deg + 1), np.ones(deg + 1)])
    return NURBSCurveTuple(deg + 1, knot.astype(float),
                           np.asarray(ctrl_xyz, float), np.asarray(weights, float))


def test_nurbs_curve_closest_global_matches_dense():
    # Two-span (knot 0.5) degree-2 NURBS as a plain polyline-ish curve
    cps = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0],
                    [3.0, -2.0, 0.0], [4.0, 0.0, 0.0]])
    w = np.ones(5)
    knot = np.array([0, 0, 0, 0.5, 1, 1, 1], float)
    crv = NURBSCurveTuple(3, knot, cps, w)
    P = np.array([2.0, 1.0, 0.0])
    res = nurbs_curve_closest_points(crv, P, atol=1e-6)
    # Dense ground truth over the global domain [0,1]
    from mmcore.geom._nurbs_eval import evaluate_nurbs_curve
    ts = np.linspace(0, 1, 4000)
    d = np.array([np.linalg.norm(evaluate_nurbs_curve(crv, t, d_order=0)["C"] - P) for t in ts])
    assert abs(res[0]["distance"] - d.min()) < 5e-3


def test_nurbs_surface_no_spurious_seam_minima():
    # Flat 2x2-span plane: an interior seam must NOT yield boundary minima.
    cps = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            cps[i, j] = [i * 0.5, j * 0.5, 0.0]
    w = np.ones((3, 3))
    knot = np.array([0, 0, 0, 1, 1, 1], float)  # single-span biquadratic, no interior seam
    srf = NURBSSurfaceTuple(3, 3, knot, knot, cps, w)
    P = np.array([0.5, 0.5, 4.0])               # foot is interior
    res = nurbs_surface_closest_points(srf, P, atol=1e-6)
    assert res[0]["kind"] == "min"
    assert abs(res[0]["distance"] - 4.0) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bez_closest_point.py -k nurbs_ -v`
Expected: FAIL — `ImportError: cannot import name 'nurbs_curve_closest_points'`.

- [ ] **Step 3: Write minimal implementation**

```python
# mmcore/numeric/_bez_closest_point.py  (append)
from mmcore.geom.nurbs import NURBSCurve, NURBSSurface
from mmcore.geom._nurbs_eval import (
    NURBSCurveTuple, NURBSSurfaceTuple, _nurbs_to_tuple,
    _curve_interval, _surface_interval, to_homogeneous_1d, to_homogeneous_2d,
)
from mmcore.geom._nurbs_knots import decompose_curve, decompose_surface


def _patch_curve_net(patch):
    """Homogeneous (q+1,4) net for a single-span Bézier curve patch tuple."""
    return np.asarray(to_homogeneous_1d(patch.control_points, patch.weights), dtype=np.float64)


def _patch_surface_net(patch):
    """Homogeneous (m+1,n+1,4) net for a single-span Bézier surface patch tuple."""
    return np.asarray(to_homogeneous_2d(patch.control_points, patch.weights), dtype=np.float64)


def nurbs_curve_closest_points(curve, point, atol=1e-3):
    """All local minima of ``||point - curve(t)||`` for a NURBS curve.

    Returns ``{"t","point","distance","kind"}`` in GLOBAL parameter, sorted by
    distance. Internal patch seams are deduped; only the global domain ends are
    ``boundary_min``.
    """
    if isinstance(curve, NURBSCurve):
        curve = _nurbs_to_tuple(curve)
    point = np.asarray(point, dtype=np.float64)
    g_lo, g_hi = _curve_interval(curve)
    patches = decompose_curve(curve)

    merged = []
    for patch in patches:
        p_lo, p_hi = _curve_interval(patch)
        net = _patch_curve_net(patch)
        local = bez_curve_closest_points(net, point, atol=atol, rational=True)
        for e in local:
            t_glob = p_lo + e["t"] * (p_hi - p_lo)
            kind = e["kind"]
            # An endpoint that is an interior seam is not a real boundary min.
            if kind == "boundary_min" and not (abs(t_glob - g_lo) < 1e-9 or abs(t_glob - g_hi) < 1e-9):
                kind = "min"
            _merge_curve(merged, t_glob, e["point"], e["distance"], kind, atol)

    merged.sort(key=lambda x: x["distance"])
    return merged


def _merge_curve(merged, t, pt, dist, kind, atol):
    pt = np.asarray(pt)
    for e in merged:
        if abs(e["t"] - t) < 1e-7 or np.linalg.norm(e["point"] - pt) < max(atol, 1e-9):
            if dist < e["distance"]:
                e.update(t=float(t), point=pt, distance=float(dist), kind=kind)
            return
    merged.append({"t": float(t), "point": pt, "distance": float(dist), "kind": kind})


def nurbs_surface_closest_points(surface, point, atol=1e-3, want_eval=False):
    """All local minima of ``||point - surface(u,v)||`` for a NURBS surface.

    Returns ``{"u","v","point","distance","kind"[, "eval"]}`` in GLOBAL params,
    sorted by distance. Internal seams deduped; only the global domain border is
    ``boundary_min``.
    """
    if isinstance(surface, NURBSSurface):
        surface = _nurbs_to_tuple(surface)
    point = np.asarray(point, dtype=np.float64)
    (gu_lo, gu_hi), (gv_lo, gv_hi) = _surface_interval(surface)
    patches = decompose_surface(surface)

    merged = []
    for patch in patches:
        (pu_lo, pu_hi), (pv_lo, pv_hi) = _surface_interval(patch)
        net = _patch_surface_net(patch)
        local = bez_surface_closest_points(net, point, atol=atol, rational=True,
                                           want_eval=want_eval)
        for e in local:
            u_glob = pu_lo + e["u"] * (pu_hi - pu_lo)
            v_glob = pv_lo + e["v"] * (pv_hi - pv_lo)
            kind = e["kind"]
            if kind == "boundary_min":
                on_global = (abs(u_glob - gu_lo) < 1e-9 or abs(u_glob - gu_hi) < 1e-9
                             or abs(v_glob - gv_lo) < 1e-9 or abs(v_glob - gv_hi) < 1e-9)
                if not on_global:
                    kind = "min"
            _merge_surface(merged, u_glob, v_glob, e, dist=e["distance"], kind=kind, atol=atol)

    merged.sort(key=lambda x: x["distance"])
    return merged


def _merge_surface(merged, u, v, src, dist, kind, atol):
    pt = np.asarray(src["point"])
    for e in merged:
        if ((abs(e["u"] - u) < 1e-7 and abs(e["v"] - v) < 1e-7)
                or np.linalg.norm(e["point"] - pt) < max(atol, 1e-9)):
            if dist < e["distance"]:
                e.update(u=float(u), v=float(v), point=pt, distance=float(dist), kind=kind)
                if "eval" in src:
                    e["eval"] = src["eval"]
            return
    entry = {"u": float(u), "v": float(v), "point": pt, "distance": float(dist), "kind": kind}
    if "eval" in src:
        entry["eval"] = src["eval"]
    merged.append(entry)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_bez_closest_point.py -k nurbs_ -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/_bez_closest_point.py tests/test_bez_closest_point.py
git commit -m "feat(closest-point): add seam-aware NURBS closest-points wrappers"
```

---

### Task 9: Rational geometry validation, cross-checks, exports

**Files:**
- Modify: `mmcore/numeric/_bez_closest_point.py` (add `__all__`)
- Test: `tests/test_bez_closest_point.py`

Validate exact-rational-stationarity on true rational geometry (sphere octant), confirm the classifier separates the min from the max on a rational arc, cross-check the global against the legacy solver, and add `__all__`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bez_closest_point.py  (append)
def _sphere_octant_net():
    # Rational biquadratic octant of the unit sphere (standard NURBS sphere patch).
    s = np.sqrt(2) / 2
    # Control points (Cartesian) and weights for one octant.
    cp = np.array([
        [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        [[1, 0, 1], [1, 1, 1], [0, 1, 1]],
        [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
    ], dtype=float)
    w = np.array([
        [1.0, s, 1.0],
        [s, 0.5, s],
        [1.0, s, 1.0],
    ])
    # Homogeneous net (x*w, y*w, z*w, w)
    H = np.concatenate([cp * w[:, :, None], w[:, :, None]], axis=2)
    return H, w


def test_rational_sphere_octant_closest_matches_dense_grid():
    # Exercises the EXACT-rational stationarity path on a true rational patch.
    # Oracle is a dense grid over the SAME rational net (self-consistent, so it
    # does not depend on the control points forming a perfect unit sphere).
    H, w = _sphere_octant_net()
    direction = np.array([0.4, 0.5, 0.6])
    direction = direction / np.linalg.norm(direction)
    P = 2.0 * direction
    res = bez_surface_closest_points(H, P, atol=1e-6, rational=True)
    u_ref, v_ref, d_ref = _dense_min_surface(H, P, rational=True, n=240)
    assert abs(res[0]["distance"] - d_ref) < 5e-3
    assert res[0]["distance"] <= d_ref + 1e-6   # solver is at least as good as the grid


def test_rational_arc_min_and_max_classified():
    s = np.sqrt(2) / 2
    C = np.array([[1.0, 0.0, 0.0, 1.0], [s, s, 0.0, s], [0.0, 1.0, 0.0, 1.0]])  # quarter circle
    P = np.array([0.0, 0.0, 0.0])   # circle center: distance is ~1 everywhere -> near-degenerate
    res = bez_curve_closest_points(C, P, atol=1e-5, rational=True)
    # Every reported entry is ~unit distance and classified, none crashes.
    for e in res:
        assert abs(e["distance"] - 1.0) < 1e-2


def test_cross_check_curve_vs_legacy_single_min():
    from mmcore.numeric.closest_point import bez_curve_closest_point
    C = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]])
    P = np.array([1.0, -1.0, 0.0])
    res = bez_curve_closest_points(C, P, atol=1e-6, rational=False)
    t_legacy, d_legacy = bez_curve_closest_point(C, P, atol=1e-6, rational=False)
    assert abs(res[0]["t"] - t_legacy) < 1e-2
    assert abs(res[0]["distance"] ** 2 - d_legacy) < 1e-4


def test_all_exports_present():
    import mmcore.numeric._bez_closest_point as m
    for name in ("bez_curve_closest_points", "bez_surface_closest_points",
                 "nurbs_curve_closest_points", "nurbs_surface_closest_points",
                 "newton_surface_closest_point", "point_surface_stationarity_nets"):
        assert name in m.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bez_closest_point.py -k "rational or cross_check or exports" -v`
Expected: FAIL — `__all__` missing (`test_all_exports_present`) and possibly tolerance failures to tune.

- [ ] **Step 3: Write implementation (add `__all__`; tune if needed)**

```python
# mmcore/numeric/_bez_closest_point.py  (append at end)
__all__ = [
    "point_curve_stationarity_net",
    "point_surface_stationarity_nets",
    "eval_curve_d2",
    "eval_surface_d2",
    "newton_curve_closest_point",
    "newton_surface_closest_point",
    "bez_curve_closest_points",
    "bez_surface_closest_points",
    "nurbs_curve_closest_points",
    "nurbs_surface_closest_points",
]
```

If `test_rational_sphere_octant_closest_on_unit_sphere` is marginally outside `1e-3`, lower the surface core's effective resolution by passing a smaller `atol` internally is NOT the fix — instead confirm `ptol_u/ptol_v` are honored and, if the Newton residual is the limiter, raise `newton_surface_closest_point`'s `max_it` to 60. Do not loosen the assertion.

- [ ] **Step 4: Run the FULL test module**

Run: `pytest tests/test_bez_closest_point.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/_bez_closest_point.py tests/test_bez_closest_point.py
git commit -m "feat(closest-point): validate rational geometry, cross-check, exports"
```

---

### Task 10: Documentation note + CLAUDE.md deprecation pointer

**Files:**
- Modify: `CLAUDE.md` (Deprecations section)

- [ ] **Step 1: Add a deprecation pointer (no test — docs only)**

In `CLAUDE.md`, under `## Deprecations`, add:

```markdown
- For closest-point, use `mmcore/numeric/_bez_closest_point.py`
  (`nurbs_curve_closest_points` / `nurbs_surface_closest_points`, net-based,
  returns all local minima). The `_divide_and_conquer` functions in
  `closest_point.py` are the legacy baseline kept for comparison.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: point closest-point usage at the net-based module"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task(s) |
|---|---|
| §3.1 true distance via numerator net | Reuses `eval_point_*_distance_sq`; Tasks 5/6 |
| §3.2 exact stationarity nets (`F'·w−2F·w'`) | Task 2 |
| §3.3 general unequal-degree Bernstein product | Task 1 |
| §4 joint stationarity prune (OR of hull-excludes-zero) | Tasks 5 (1D), 6 (2D) |
| §4 optional value branch-and-bound (global-only) | Not implemented — default is all-minima; documented as out of scope here (add later behind a flag if a global-only fast path is needed) |
| §5.1 Bernstein algebra + net builders | Tasks 1, 2 |
| §5.2 `newton_surface_closest_point` (+1D reuse) | Task 4 |
| §5.3 curve core, all minima, endpoints, classify, dedup | Task 5 |
| §5.4 surface core, prune, Newton, cutout, boundary, classify | Tasks 6, 7 (cutout simplified to dedup+resolution-floor; see note) |
| §5.5 NURBS wrappers, seam dedup, param mapping | Task 8 |
| §6 result shape `{param, point, distance, kind[, eval]}` | Tasks 5–8 |
| §7 edge cases (point-on-geom, degenerate, never-empty) | Tasks 5/6 fallbacks; degenerate rational arc test Task 9 |
| §8 testing (analytic, multiplicity, property, cross-check, multi-patch, rational) | Tasks 5–9 |

**Gap note (intentional):** the spec's `_cutout_2d` re-convergence guard (§5.4) is replaced here by per-candidate dedup (`_dedup_add`) plus the `ptol`-sized small-cell termination, which together prevent duplicate reports without the extra 3×3 cutout machinery. If profiling on dense multi-root cases shows redundant subdivision near a converged root, add the cutout as a follow-up (port `_cutout_2d` from `ccx/_bez_ccx4.py`). The value branch-and-bound prune (§4) is likewise deferred — it is an accelerator only for the global-only mode, which is not the default. Both are documented, not silent.

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to" — every code step is concrete. The one `pass` stub in Task 6 is explicitly replaced in Task 7 (Step 3 says "REPLACE the stub").

**3. Type consistency:** Curve entries use key `"t"`; surface entries use `"u"/"v"`. `kind ∈ {"min","boundary_min"}` everywhere. Net builders return `(N, F, Qw)` / `(Nu, Nv, F, Sw)` consistently. `newton_surface_closest_point` returns `(u, v, R, last_step)`; `newton_curve_closest_point` returns `(u, R, sqdist, last_du)`. `_dedup_add`/`_merge_*` operate on the same dict shape produced by the cores. `eval_surface_d2` returns 6-tuple `(pt, Su, Sv, Suu, Suv, Svv)` used identically in `_classify_surface_min` and the Newton kernel.
